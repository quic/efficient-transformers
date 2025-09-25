# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import torch.nn as nn
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel

from .attention import QEffJointTransformerBlock


class QEffSD3Transformer2DModel(SD3Transformer2DModel):
    def __qeff_init__(self):
        self.transformer_blocks = nn.ModuleList()
        self._block_classes = set()

        for i in range(self.config.num_layers):
            block_class_name = f"QEffJointTransformerBlock_{i:02d}"
            BlockClass = type(block_class_name, (QEffJointTransformerBlock,), {})
            block = BlockClass(
                dim=self.inner_dim,
                num_attention_heads=self.config.num_attention_heads,
                attention_head_dim=self.config.attention_head_dim,
                context_pre_only=i == self.config.num_layers - 1,
                qk_norm=self.config.qk_norm,
                use_dual_attention=True if i in self.dual_attention_layers else False,
            )
            self.transformer_blocks.append(block)
            self._block_classes.add(BlockClass)
