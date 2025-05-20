# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from QEfficient.customop import CtxGatherFuncCB


class LinearMultiLoRA(nn.Linear):
    def multilora_init(self, lora_rank, max_num_adapters):
        if lora_rank < 1 or max_num_adapters < 1:
            raise ValueError("lora_rank and max_num_adapters must be greater or equal to 1")

        self.max_num_adapters = max_num_adapters
        self.lora_rank = lora_rank

        self.lora_a_weights = nn.Parameter(
            self.weight.new_zeros(self.max_num_adapters + 1, 1, self.in_features, self.lora_rank)
        )
        self.lora_a_weights.requires_grad = False
        self.lora_b_weights = nn.Parameter(
            self.weight.new_zeros(self.max_num_adapters + 1, 1, self.lora_rank, self.out_features)
        )
        self.lora_b_weights.requires_grad = False
        self.lora_scalings = torch.full((self.max_num_adapters + 1, 1, 1, 1), 1.0, dtype=torch.float)

        nn.init.kaiming_uniform_(self.lora_a_weights, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_weights)

    def forward(self, x: torch.Tensor, lora_ids: torch.Tensor):
        result = F.linear(x, self.weight, bias=self.bias)

        # multilora implementation: lora_ids <batch_size, 1>
        other_indices_a = torch.arange(self.lora_a_weights.shape[2]).view(1, 1, -1)
        selected_lora_a_weights = CtxGatherFuncCB.apply(
            self.lora_a_weights, lora_ids, other_indices_a
        )  # <num_loras, 1, feature, r>
        other_indices_b = torch.arange(self.lora_b_weights.shape[2]).view(1, 1, -1)
        selected_lora_b_weights = CtxGatherFuncCB.apply(
            self.lora_b_weights, lora_ids, other_indices_b
        )  # <num_loras, 1, r, feature>
        other_indices_s = torch.arange(self.lora_scalings.shape[2]).view(1, 1, -1)
        selected_lora_scalings = CtxGatherFuncCB.apply(
            self.lora_scalings, lora_ids, other_indices_s
        )  # <num_loras, 1, 1, 1>

        selected_lora_a_weights = selected_lora_a_weights.squeeze(1)
        selected_lora_b_weights = selected_lora_b_weights.squeeze(1)
        selected_lora_scalings = selected_lora_scalings.squeeze(1)

        result = result + x @ selected_lora_a_weights @ selected_lora_b_weights * selected_lora_scalings

        return result


class LinearBase(nn.Linear):
    def forward(self, x: torch.Tensor, **kwargs: Any):
        return super().forward(x)
