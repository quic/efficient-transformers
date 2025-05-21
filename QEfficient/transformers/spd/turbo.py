# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch

from QEfficient.utils.checkpoint_utils import load_checkpoint


class ResBlock(torch.nn.Module):
    """
    A Residual Block module.
    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.
    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = torch.nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


def post_process_turbo_state_dict(state_dict: dict) -> dict:
    """normaize turbo state dict keys
    Args:
        state_dict (dict): turbo state dict
    Returns:
        dict: normalized state dict
    """
    new_state_dict = dict()
    for name, weights in state_dict.items():
        new_name = name.replace("projections.", "")
        new_state_dict[new_name] = weights
    return new_state_dict


def build_and_attach_turbo(model, speculative_config: dict, speculative_weights: str):
    """build and attach turbo projections
    Args:
        model: model to attach projections to
        speculative_config (dict): speculative config file used to build projections
    Returns:
        model: model with turbo projections
    """
    hidden_size = model.config.hidden_size
    num_layers = speculative_config["turbo_num_layers"]
    num_heads = speculative_config["turbo_num_heads"]
    projections = torch.nn.ModuleList(
        [
            torch.nn.Sequential(
                *([ResBlock(hidden_size)] * num_layers),
            )
            for _ in range(num_heads)
        ],
    )
    load_checkpoint(projections, speculative_weights, strict=True, post_process_func=post_process_turbo_state_dict)
    model.projections = projections
    speculative_config["num_speculative_tokens"] = num_heads
    return model
