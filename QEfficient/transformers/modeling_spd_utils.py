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
    num_speculative_tokens: Optional[int] = None,
) -> torch.Tensor:
    """
    Filter hidden states based on whether this is a TLM SpD model

    ``Mandatory`` Args:
        :hidden_states (torch.Tensor): Hidden states tensor.
        :position_ids (torch.Tensor): Position ids tensor.
    ``Optional`` Args:
        :num_speculative_tokens (int, optional): Number of speculative tokens, specified only for TLM SpD model

    Returns:
        :torch.Tensor: Filtered hidden states.
    """
    batch_indices = torch.arange(position_ids.shape[0])
    if num_speculative_tokens is not None:
        # all logits need to be computed
        return hidden_states[batch_indices].squeeze(1)
    # Cast to INT32 to avoid issue while running in ONNXRT
    logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
    return hidden_states[batch_indices.view(-1, 1), logit_index]
