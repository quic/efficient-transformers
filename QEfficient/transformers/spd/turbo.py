# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
from accelerate import load_checkpoint_and_dispatch


class ResBlock(torch.nn.Module):  # Res block for Turbo LoRA projection heads
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


def build_and_attach_turbo(model, speculative_config):
    hidden_size = model.config.hidden_size["hidden_size"]
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
    model.projections = projections
    speculative_weights = speculative_config["speculative_weights"]
    model = load_checkpoint_and_dispatch(model, checkpoint=speculative_weights, strict=False)
    speculative_config["num_speculative_tokens"] = num_heads
    return model
