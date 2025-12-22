# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
from typing import Optional, Tuple

import torch
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle


class QEffAdaLayerNormZero(AdaLayerNormZero):
    def forward(
        self,
        x: torch.Tensor,
        shift_msa: Optional[torch.Tensor] = None,
        scale_msa: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x


class QEffAdaLayerNormZeroSingle(AdaLayerNormZeroSingle):
    def forward(
        self,
        x: torch.Tensor,
        scale_msa: Optional[torch.Tensor] = None,
        shift_msa: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x


class QEffAdaLayerNormContinuous(AdaLayerNormContinuous):
    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = conditioning_embedding
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x
