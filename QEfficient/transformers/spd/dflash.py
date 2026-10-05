# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch


def compute_dflash_target_hidden_states(
    target_hidden_list: list[torch.Tensor],
    fc: torch.nn.Module,
    hidden_norm: torch.nn.Module,
) -> torch.Tensor:
    """
    Build the DFlash TLM target hidden state from the hidden states collected at
    ``target_layer_ids``: concatenate them along the last dim, project with ``fc``,
    then normalize with ``hidden_norm``.

    ``Mandatory`` Args:
        :target_hidden_list (list[torch.Tensor]): Hidden states collected at each target layer.
        :fc (torch.nn.Module): Linear projection from concatenated target hidden states to hidden_size.
        :hidden_norm (torch.nn.Module): Norm applied after the fc projection.

    Returns:
        :torch.Tensor: The projected and normalized target hidden state.
    """
    target_hidden = torch.cat(target_hidden_list, dim=-1)
    return hidden_norm(fc(target_hidden))
