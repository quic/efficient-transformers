# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
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
        self.max_num_adapters = max_num_adapters
        self.lora_rank = lora_rank

        self.lora_weight_A = nn.Parameter(
            self.weight.new_zeros(self.max_num_adapters, 1, self.in_features, self.lora_rank)
        )
        self.lora_weight_A.requires_grad = False
        self.lora_weight_B = nn.Parameter(
            self.weight.new_zeros(self.max_num_adapters, 1, self.lora_rank, self.out_features)
        )
        self.lora_weight_B.requires_grad = False
        self.lora_weight_C = torch.full((self.max_num_adapters, 1, 1, 1), 1.0, dtype=torch.float)

        nn.init.kaiming_uniform_(self.lora_weight_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_weight_B)

    def forward(self, x: torch.Tensor, **kwargs: Any):
        lora_ids = kwargs.pop("lora_ids", torch.zeros((x.shape[0]), dtype=torch.int64).view(-1, 1))

        with torch.no_grad():
            result = F.linear(x, self.weight, bias=self.bias)

            # multilora implementation: lora_ids <batch_size, 1>
            other_indices_A = torch.arange(self.lora_weight_A.shape[2]).view(1, 1, -1)
            A_embedding = CtxGatherFuncCB.apply(
                self.lora_weight_A, lora_ids, other_indices_A
            )  # <num_loras, 1, feature, r>
            other_indices_B = torch.arange(self.lora_weight_B.shape[2]).view(1, 1, -1)
            B_embedding = CtxGatherFuncCB.apply(
                self.lora_weight_B, lora_ids, other_indices_B
            )  # <num_loras, 1, r, feature>
            other_indices_C = torch.arange(self.lora_weight_C.shape[2]).view(1, 1, -1)
            C_embedding = CtxGatherFuncCB.apply(self.lora_weight_C, lora_ids, other_indices_C)  # <num_loras, 1, 1, 1>

            A_embedding = A_embedding.squeeze(1)
            B_embedding = B_embedding.squeeze(1)
            C_embedding = C_embedding.squeeze(1)

            result = result + x @ A_embedding @ B_embedding * C_embedding

            return result


class LinearBase(nn.Linear):
    def forward(self, x: torch.Tensor, **kwargs: Any):
        return super().forward(x)
