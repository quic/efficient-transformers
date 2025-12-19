# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import inspect
from functools import wraps
from typing import Callable, Optional

import torch
import torch.nn as nn
from transformers.masking_utils import (
    ALL_MASK_ATTENTION_FUNCTIONS,
    _ignore_causal_mask_sdpa,
    and_masks,
    causal_mask_function,
    eager_mask,
    padding_mask_function,
    prepare_padding_mask,
    sdpa_mask,
)


def mean_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Performs mean pooling on the last hidden states of a transformer model.

    Args:
        last_hidden_states (torch.Tensor): The last hidden states of the transformer model.
        attention_mask (torch.Tensor): The attention mask used to mask out padding tokens.

    Returns:
        torch.Tensor: The mean pooled last hidden states.
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    return torch.sum(last_hidden_states * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Performs average pooling on the last hidden states of a transformer model.

    Args:
        last_hidden_states (torch.Tensor): The last hidden states of the transformer model.
        attention_mask (torch.Tensor): The attention mask used to mask out padding tokens.

    Returns:
        torch.Tensor: The average pooled last hidden states.
    """
    last_hidden = last_hidden_states[0].masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def max_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Performs max pooling on the last hidden states of a transformer model.

    Args:
        last_hidden_states (torch.Tensor): The last hidden states of the transformer model.
        attention_mask (torch.Tensor): The attention mask used to mask out padding tokens.

    Returns:
        torch.Tensor: The max pooled last hidden states.
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    last_hidden_states[input_mask_expanded == 0] = -1e9
    return torch.max(last_hidden_states, 1)[0]


def cls_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Performs CLS pooling on the last hidden states of a transformer model.

    Args:
        last_hidden_states (torch.Tensor): The last hidden states of the transformer model.
        attention_mask (torch.Tensor): The attention mask used to mask out padding tokens.

    Returns:
        torch.Tensor: The CLS pooled last hidden states.
    """
    return last_hidden_states[:, 0]


POOLING_MAP = {
    "mean": mean_pooling,
    "avg": average_pool,
    "cls": cls_pooling,
    "max": max_pooling,
}


class PooledModel(nn.Module):
    """
    Adds pooling functionality to embedding model.
    """

    def __init__(self, base_model, pooling_fn):
        super().__init__()
        self.config = base_model.config
        self.base_model = base_model
        self.pooling_fn = pooling_fn

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        output = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **kwargs
        )
        return self.pooling_fn(output[0], attention_mask)


def validate_user_pooling_function(user_function):
    """
    Validate a user-provided pooling function to ensure it meets the required interface.

    The function should take two arguments:
    - last_hidden_states (torch.Tensor): The last hidden states of the model.
    - attention_mask (torch.Tensor): The attention mask of the input sequence.

    It should return a torch.Tensor representing the pooled output.

    Args:
        user_function (callable): The user-provided pooling function.

    Raises:
        ValueError: If the user-provided function does not meet the required interface.
    """

    if not callable(user_function):
        raise TypeError("Provided pooling function is not callable.")

    sig = inspect.signature(user_function)
    required_args = {"last_hidden_states", "attention_mask"}
    if not required_args.issubset(sig.parameters.keys()):
        raise ValueError(f"Pooling function must accept arguments: {required_args}")
    return user_function


# Custom vectorized implementation of sdpa_mask without using vmap
def sdpa_mask_without_vmap(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable | None = None,
    attention_mask: torch.Tensor | None = None,
    local_size: int | None = None,
    allow_is_causal_skip: bool = True,
    **kwargs,
) -> torch.Tensor | None:
    if mask_function is None:
        mask_function = causal_mask_function

    q_length = cache_position.shape[0]
    # Potentially pad the 2D mask, and slice it correctly
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)

    # Under specific conditions, we can avoid materializing the mask, instead relying on the `is_causal` argument
    if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
        return None

    # Potentially add the padding 2D mask
    if padding_mask is not None:
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    # Create broadcatable indices
    device = cache_position.device
    q_indices = cache_position[None, None, :, None]
    head_indices = torch.arange(1, dtype=torch.long, device=device)[None, :, None, None]
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)[:, None, None, None]
    kv_indices = torch.arange(kv_length, dtype=torch.long, device=device)[None, None, None, :] + kv_offset
    # Apply mask function element-wise through broadcasting
    causal_mask = mask_function(batch_indices, head_indices, q_indices, kv_indices)
    # Expand the mask to match batch size and query length if they weren't used in the mask function
    causal_mask = causal_mask.expand(batch_size, -1, q_length, kv_length)

    return causal_mask


# Adapted from https://github.com/huggingface/transformers/blob/v4.53.0/src/transformers/masking_utils.py#L433
def eager_mask_without_vmap(*args, **kwargs) -> torch.Tensor:
    kwargs.pop("allow_is_causal_skip", None)
    dtype = kwargs.get("dtype", torch.float32)
    mask = sdpa_mask_without_vmap(*args, allow_is_causal_skip=False, **kwargs)
    mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), torch.finfo(dtype).min)
    return mask


def register_mask_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", sdpa_mask_without_vmap)
        ALL_MASK_ATTENTION_FUNCTIONS.register("eager", eager_mask_without_vmap)
        # Call the function for loading quantized models here
        out = func(*args, **kwargs)
        ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", sdpa_mask)
        ALL_MASK_ATTENTION_FUNCTIONS.register("eager", eager_mask)
        return out

    return wrapper
