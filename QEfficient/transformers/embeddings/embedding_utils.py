# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import inspect
from typing import Optional

import torch
import torch.nn as nn


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
        self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, **kwargs
    ):
        output = self.base_model(input_ids, attention_mask, **kwargs)
        return self.pooling_fn(output[0], attention_mask)


def embedding_forward(
    self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, **kwargs
):
    print("Forward swapped with new one")
    output = self.old_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    return output[0]


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
