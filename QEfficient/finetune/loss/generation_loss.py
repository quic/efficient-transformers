# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Optional

import torch
import torch.nn as nn
from transformers.loss.loss_utils import fixed_cross_entropy

from QEfficient.finetune.loss.common import BaseLoss


class ForCausalLMLoss(BaseLoss):
    def __init__(self):
        pass

    def __call__(
        self,
        logits,
        labels,
        vocab_size: int,
        num_items_in_batch: Optional[torch.Tensor] = None,
        loss_weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        shift_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()

        if shift_labels is None:
            # Shift so that tokens < n predict n
            labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()
            shift_labels = shift_labels.to(logits.device)

        if loss_weight is None:
            # Flatten the tokens
            logits = logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
            return loss
        else:
            total_loss = torch.tensor(0.0, device=logits.device)
            bs = logits.shape[0]
            for i in range(bs):
                # Flatten the tokens
                _logits = logits[i].view(-1, vocab_size)
                _shift_labels = shift_labels[i].view(-1)
                # Enable model parallelism
                loss = fixed_cross_entropy(_logits, _shift_labels, ignore_index=ignore_index, **kwargs)
                loss *= loss_weight[i]
                total_loss += loss

            if torch.sum(loss_weight) == 0:
                return total_loss
            else:
                total_loss /= torch.sum(loss_weight)
                return total_loss
