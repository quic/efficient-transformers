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
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        shift_msa: Optional[torch.Tensor] = None,
        scale_msa: Optional[torch.Tensor] = None,
        # emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # if self.emb is not None:
        #     emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        # emb = self.linear(self.silu(emb))
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x


class QEffAdaLayerNormZeroSingle(AdaLayerNormZeroSingle):
    def forward(
        self,
        x: torch.Tensor,
        scale_msa: Optional[torch.Tensor] = None,
        shift_msa: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x


class QEffAdaLayerNormContinuous(AdaLayerNormContinuous):
    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        # emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        emb = conditioning_embedding
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x
