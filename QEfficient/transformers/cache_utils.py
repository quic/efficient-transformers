# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache, EncoderDecoderCache, HybridCache

from QEfficient.customop import (
    CtxGatherFunc,
    CtxGatherFunc3D,
    CtxGatherFuncCB,
    CtxGatherFuncCB3D,
    CtxScatterFunc,
    CtxScatterFunc3D,
    CtxScatterFuncCB,
    CtxScatterFuncCB3D,
)


class QEffDynamicCache(DynamicCache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    - Optimized implementation for the Cloud AI 100 to reuse KV Cache.
    - get the position_ids input using kwargs.
    - Use custom Onnxscript ops to write optimized version to generate Onnx model.

    """

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
        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            position_ids = cache_kwargs.get("position_ids")
            batch_index = cache_kwargs.get("batch_index", None)

            # Scatter
            if batch_index is not None:
                invalid_scatter_index = torch.iinfo(torch.int32).max
                scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)

                self.key_cache[layer_idx] = CtxScatterFuncCB.apply(
                    self.key_cache[layer_idx], batch_index, scatter_position_ids, key_states
                )
                self.value_cache[layer_idx] = CtxScatterFuncCB.apply(
                    self.value_cache[layer_idx], batch_index, scatter_position_ids, value_states
                )
            else:
                self.key_cache[layer_idx] = CtxScatterFunc.apply(self.key_cache[layer_idx], position_ids, key_states)
                self.value_cache[layer_idx] = CtxScatterFunc.apply(
                    self.value_cache[layer_idx], position_ids, value_states
                )

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
        k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index", None)
        ctx_len = k_out.shape[2]
        ctx_indices = torch.arange(ctx_len)[None, None, ...]
        gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
        invalid_mask = ctx_indices > gather_limit

        if torch.onnx.is_in_onnx_export():
            invalid_idx_value = torch.iinfo(torch.int32).max
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

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            k_out, v_out = key_states, value_states
        else:
            position_ids = cache_kwargs.get("position_ids")
            batch_index = cache_kwargs.get("batch_index", None)  # Check and fetch batch index value form the kwargs

            # Scatter
            if batch_index is not None:
                invalid_scatter_index = torch.iinfo(torch.int32).max
                scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)

                self.key_cache[layer_idx] = CtxScatterFuncCB.apply(
                    self.key_cache[layer_idx], batch_index, scatter_position_ids, key_states
                )

                self.value_cache[layer_idx] = CtxScatterFuncCB.apply(
                    self.value_cache[layer_idx], batch_index, scatter_position_ids, value_states
                )
            else:
                self.key_cache[layer_idx] = CtxScatterFunc.apply(self.key_cache[layer_idx], position_ids, key_states)
                self.value_cache[layer_idx] = CtxScatterFunc.apply(
                    self.value_cache[layer_idx], position_ids, value_states
                )

            k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]

            # Gather
            ctx_len = k_out.shape[2]
            ctx_indices = torch.arange(ctx_len)[None, None, ...]
            gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
            invalid_mask = ctx_indices > gather_limit

            if torch.onnx.is_in_onnx_export():
                invalid_idx_value = torch.iinfo(torch.int32).max
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

    def update3D(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            k_out, v_out = key_states, value_states
        else:
            position_ids = cache_kwargs.get("position_ids")
            batch_index = cache_kwargs.get("batch_index", None)

            if batch_index is not None:
                invalid_scatter_index = torch.iinfo(torch.int32).max
                scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)

                self.key_cache[layer_idx] = CtxScatterFuncCB3D.apply(
                    self.key_cache[layer_idx], batch_index, scatter_position_ids, key_states
                )

                self.value_cache[layer_idx] = CtxScatterFuncCB3D.apply(
                    self.value_cache[layer_idx], batch_index, scatter_position_ids, value_states
                )

            else:
                self.key_cache[layer_idx] = CtxScatterFunc3D.apply(self.key_cache[layer_idx], position_ids, key_states)
                self.value_cache[layer_idx] = CtxScatterFunc3D.apply(
                    self.value_cache[layer_idx], position_ids, value_states
                )
            k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]

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

    def _sliding_update(
        self,
        layer_idx,
        key_states,
        value_states,
        position_ids,
        batch_index,
        k_out,
        v_out,
    ):
        N = self.key_cache[layer_idx].shape[2]

        # Update the position_ids to handle the sliding window
        kv_position_ids = torch.where(position_ids == -1, position_ids, position_ids % (N - 1))
        kv_position_ids = torch.where(position_ids.max() >= (N - 1) * 2, (position_ids + 1) % N, kv_position_ids)

        # Update the cache
        self.key_cache[layer_idx] = CtxScatterFunc.apply(self.key_cache[layer_idx], kv_position_ids, key_states)
        self.value_cache[layer_idx] = CtxScatterFunc.apply(self.value_cache[layer_idx], kv_position_ids, value_states)

        k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]

        # Original Gather
        ctx_len = min(N, k_out.shape[2])
        ctx_indices = torch.arange(ctx_len)[None, None, ...]
        gather_limit = kv_position_ids.max(1, keepdim=True).values.unsqueeze(1)
        invalid_mask = ctx_indices > gather_limit
        if torch.onnx.is_in_onnx_export():
            invalid_idx_value = torch.iinfo(torch.int32).max
        else:
            invalid_idx_value = 0
        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

        # rolling indices
        all_indices = torch.arange(N) + kv_position_ids.max() + 1
        rolling_indices = torch.where(all_indices > N - 1, all_indices % N, all_indices)

        final_indices = torch.where(position_ids.max() >= (N - 1), rolling_indices, ctx_indices)

        k_out = CtxGatherFunc.apply(k_out, final_indices)
        v_out = CtxGatherFunc.apply(v_out, final_indices)
        prefill_v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)

        # Handle the rolling indices
        v_out = torch.where(position_ids.max() >= (N - 1), v_out, prefill_v_out)
        return k_out, v_out

    def _static_update(
        self,
        layer_idx,
        key_states,
        value_states,
        position_ids,
        batch_index,
        k_out,
        v_out,
    ):
        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            k_out, v_out = key_states, value_states
        else:
            # Scatter
            if batch_index is not None:
                invalid_scatter_index = torch.iinfo(torch.int32).max
                scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)

                self.key_cache[layer_idx] = CtxScatterFuncCB.apply(
                    self.key_cache[layer_idx], batch_index, scatter_position_ids, key_states
                )

                self.value_cache[layer_idx] = CtxScatterFuncCB.apply(
                    self.value_cache[layer_idx], batch_index, scatter_position_ids, value_states
                )
            else:
                self.key_cache[layer_idx] = CtxScatterFunc.apply(self.key_cache[layer_idx], position_ids, key_states)
                self.value_cache[layer_idx] = CtxScatterFunc.apply(
                    self.value_cache[layer_idx], position_ids, value_states
                )

            k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]

            # Gather
            ctx_len = k_out.shape[2]
            ctx_indices = torch.arange(ctx_len)[None, None, ...]
            gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
            invalid_mask = ctx_indices > gather_limit

            if torch.onnx.is_in_onnx_export():
                invalid_idx_value = torch.iinfo(torch.int32).max
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

    def update_hybrid_chunked(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates cache with support for both sliding window and position-based updates.
        """
        if cache_kwargs is None:
            cache_kwargs = {}

        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        key_states = key_states.to(k_out.dtype)
        value_states = value_states.to(v_out.dtype)

        # Get cache parameters
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index", None)
        sliding_window = cache_kwargs.get("is_sliding", None)

        if sliding_window[layer_idx]:
            update_fn = self._sliding_update
        else:
            update_fn = self._static_update

        k_out, v_out = update_fn(
            layer_idx,
            key_states,
            value_states,
            position_ids,
            batch_index,
            k_out,
            v_out,
        )

        return k_out, v_out


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


class QEffHybridCache(HybridCache):
    def __init__(self, config, batch_size, max_cache_len):
        super().__init__(config, batch_size, max_cache_len=max_cache_len)
        # breakpoint()
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
        # breakpoint()
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
            N = self.key_cache[layer_idx].shape[2]
            kv_position_ids = torch.where(
                (~is_sliding_layer | (position_ids == -1)), position_ids, position_ids % (N - 1)
            )

            kv_position_ids = torch.where(
                is_sliding_layer & (position_ids.max() >= (N - 1) * 2), (position_ids + 1) % N, kv_position_ids
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
            ctx_len = self.key_cache[layer_idx].shape[2]
            ctx_indices = torch.arange(ctx_len)[None, None, ...]
            gather_limit = kv_position_ids.max(1, keepdim=True).values.unsqueeze(1)
            invalid_mask = ctx_indices > gather_limit
            if torch.onnx.is_in_onnx_export():
                invalid_idx_value = torch.iinfo(torch.int32).max
            else:
                invalid_idx_value = 0
            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

            all_indices = torch.arange(N) + kv_position_ids.max() + 1
            rolling_indices = torch.where(all_indices > N - 1, all_indices % N, all_indices)
            final_indices = torch.where(
                (is_sliding_layer & (position_ids.max() >= (N - 1))), rolling_indices, ctx_indices
            )
            k_out = CtxGatherFunc.apply(k_out, final_indices)
            v_out = CtxGatherFunc.apply(v_out, final_indices)
            ctx_v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)
            v_out = torch.where((is_sliding_layer & (position_ids.max() >= (N - 1))), v_out, ctx_v_out)
        return k_out, v_out
