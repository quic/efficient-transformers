# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


from QEfficient.finetune.loss.generation_loss import ForCausalLMLoss
from QEfficient.finetune.loss.seq_cls_loss import ForSequenceClassificationLoss


loss_fn_dict = {
    "seq_classification": ForSequenceClassificationLoss,
    "generation": ForCausalLMLoss,
}


def get_loss(task_name: str):
    if task_name not in loss_fn_dict:
        raise RuntimeError(f"No loss function registered for this task name: '{task_name}'.")
    return loss_fn_dict[task_name]
