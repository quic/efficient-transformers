# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Optional

import torch
from torch.nn import BCEWithLogitsLoss, MSELoss
from transformers.loss.loss_utils import fixed_cross_entropy

from QEfficient.finetune.loss.common import BaseLoss

# Note: Below code is taken from https://github.com/huggingface/transformers/blob/2166b6b4ff09f6dd3867ab982f262f66482aa968/src/transformers/loss/loss_utils.py#L92
#       The original code is modified to take loss_weight into consideration.
#       It will apply a boolean value to the loss for each item in the batch.
#       This is helpful when we explicitly want to set loss for a particular
#       sample in batch to zero. E.g. when padding of dataset is done.


class ForSequenceClassificationLoss(BaseLoss):
    def __init__(self, num_labels):
        self.num_labels = num_labels

    def __call__(
        self, pooled_logits: torch.Tensor, labels: torch.Tensor, loss_weight: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        num_labels = self.num_labels
        if num_labels == 1:
            problem_type = "regression"
        elif num_labels > 1 and (labels.dtype in (torch.long, torch.int)):
            problem_type = "single_label_classification"
        else:
            problem_type = "multi_label_classification"

        labels = labels.to(pooled_logits.device)
        if problem_type == "regression":
            loss_fct = MSELoss()
            if num_labels == 1:
                return loss_fct(pooled_logits.squeeze(), labels.squeeze())
            else:
                return loss_fct(pooled_logits, labels)
        if problem_type == "single_label_classification":
            if loss_weight is None:
                return fixed_cross_entropy(pooled_logits.view(-1, num_labels), labels.view(-1), **kwargs)
            else:
                total_loss = torch.tensor(0.0, device=pooled_logits.device)
                bs = pooled_logits.shape[0]
                for i in range(bs):
                    total_loss += loss_weight[i] * fixed_cross_entropy(
                        pooled_logits[i].view(-1, num_labels), labels[i].view(-1), **kwargs
                    )
                if torch.sum(loss_weight) == 0:
                    return total_loss
                else:
                    total_loss /= torch.sum(loss_weight)
                    return total_loss

        if problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            return loss_fct(pooled_logits, labels)
