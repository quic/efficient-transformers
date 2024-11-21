# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Optional

import torch


def filter_hidden_states(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    num_logits_to_keep: Optional[int] = None,
) -> torch.Tensor:
    """
    Filter hidden states based on whether this is a TLM SpD model

    ``Mandatory`` Args:
        :hidden_states (torch.Tensor): Hidden states tensor.
        :position_ids (torch.Tensor): Position ids tensor.
    ``Optional`` Args:
        :num_logits_to_keep (int, optional): Number of speculative tokens, specified only for TLM SpD model

    Returns:
        :torch.Tensor: Filtered hidden states.
    """
    batch_size = position_ids.size(0)
    batch_indices = torch.arange(batch_size)
    # Cast to INT32 to avoid issue while running in ONNXRT
    logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
    if num_logits_to_keep is None:
        # return the last logit
        return hidden_states[batch_indices.view(-1, 1), logit_index]
    # gather approach
    lower_idx = torch.where(logit_index < num_logits_to_keep, 0, logit_index+1 - num_logits_to_keep).view(-1,1) # shape: [bsz, 1]
    spec_idx = torch.arange(num_logits_to_keep).view(1,-1) # shape: [1, k]
    indices = torch.add(lower_idx, spec_idx).unsqueeze(2) # shape: [bsz, k, 1]
    indices = indices.repeat(1, 1, hidden_states.size(-1)) # shape: [bsz, ,k, d_model]
    hidden_states = torch.gather(hidden_states, dim=1, index=indices) # shape: [bsz, k, d_model]
    return hidden_states
