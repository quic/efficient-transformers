# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
import torch
from torch import nn


class QeffProxyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.embed_tokens = None
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, hidden_states):
        inputs_embeds = torch.unsqueeze(hidden_states.float(), 2).expand(-1, -1, self.embedding_dim)
        return inputs_embeds


class QeffProxyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        self.lm_head = None

    def forward(self, hidden_states):
        return hidden_states
