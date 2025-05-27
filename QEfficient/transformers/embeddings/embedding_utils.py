# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from logging import warning
from typing import Optional

import torch
import torch.nn as nn


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
# def mean_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#     # Apply the attention mask to the hidden states
#     masked_hidden = last_hidden_states[0] * attention_mask[..., None]
    
#     # Sum the masked hidden states along the sequence dimension
#     sum_hidden = masked_hidden.sum(dim=1)
    
#     # Count the number of valid (non-masked) tokens
#     valid_token_count = attention_mask.sum(dim=1)[..., None]
    
#     # Compute the mean by dividing summed hidden states by the count of valid tokens
#     return sum_hidden / valid_token_count


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states[0].masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def max_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  
    return torch.max(token_embeddings, 1)[0]


def cls_pooling(token_embeddings, attention_mask):
    return token_embeddings[:, 0]


POOLING_MAP = {
    "mean": mean_pooling,
    "avg": average_pool,
    "cls": cls_pooling,
    "max": max_pooling,
}


class PooledModel(nn.Module):
    def __init__(self, base_model, pooling_fn):
        super().__init__()
        self.config = base_model.config
        self.base_model = base_model
        self.pooling_fn = pooling_fn

    def forward(
        self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, **kwargs
    ):
        warning("")
        output = self.base_model(input_ids, attention_mask, **kwargs)
        return self.pooling_fn(output[0], attention_mask)


def embedding_transform(func):
    def wrapper(self, model, **kwargs):
        if kwargs.get("pooling") is not None:
            pooling = kwargs["pooling"]
            pooling_method = POOLING_MAP[pooling]
            model = PooledModel(model, pooling_method)
        result = func(self, model, **kwargs)
        return result

    return wrapper
