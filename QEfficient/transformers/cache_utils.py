# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


from typing import Any, Dict, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

from QEfficient.customop import CtxGatherFunc, CtxScatterFunc


class QEffDynamicCache(DynamicCache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    - Optimized implementation for the Cloud AI 100 to reuse KV Cache.
    - get the position_ids input using kwargs.
    - Use custom Onnxscript ops to write optimized version to generate Onnx model.

    """

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
            batch_index = cache_kwargs.get("batch_index")
            position_ids = cache_kwargs.get("position_ids")
            ctx_len = self.key_cache[layer_idx].shape[2]

            # Scatter
            invalid_scatter_index = ctx_len - 1  # remove post the backend compiler fix is in mainline sdk
            scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)
            self.key_cache[layer_idx] = CtxScatterFunc.apply(
                self.key_cache[layer_idx], batch_index, scatter_position_ids, key_states
            )
            self.value_cache[layer_idx] = CtxScatterFunc.apply(
                self.value_cache[layer_idx], batch_index, scatter_position_ids, value_states
            )
            k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]

            # Gather
            ctx_indices = torch.arange(ctx_len)[None, None, ...]
            gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
            invalid_mask = ctx_indices > gather_limit
            if torch.onnx.is_in_onnx_export():
                # remove post the backend compiler fix is in mainline sdk
                # invalid_gather_idx = torch.iinfo(torch.int32).max
                invalid_gather_idx = ctx_len - 1
            else:
                invalid_gather_idx = 0
            # invalid_gather_idx = torch.iinfo(torch.int32).max
            ctx_indices = torch.where(invalid_mask, invalid_gather_idx, ctx_indices)
            k_out = CtxGatherFunc.apply(k_out, batch_index, ctx_indices)
            v_out = CtxGatherFunc.apply(v_out, batch_index, ctx_indices)
            v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)

        return k_out, v_out
