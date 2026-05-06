# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import logging

from trl import DPOConfig, DPOTrainer

from QEfficient.finetune.experimental.core.component_registry import registry
from QEfficient.finetune.experimental.core.config_manager import PeftConfig

logger = logging.getLogger(__name__)


@registry.trainer_module(
    name="dpo",
    args_cls=DPOConfig,
    required_kwargs={"peft_config": PeftConfig, "ref_model": None},
)
class DPOTrainerModule(DPOTrainer):
    """DPO Trainer that disables DataParallel (single-device, PP, or DDP only).

    The ``compute_loss_func`` kwarg forwarded by the pipeline is silently
    dropped because ``DPOTrainer`` computes its own loss and does not accept
    that argument.

    After the parent ``__init__`` tokenizes the dataset, any sample whose
    ``chosen_ids`` or ``rejected_ids`` is empty (e.g. because a very long
    prompt consumed the entire ``max_length`` budget during truncation) is
    removed.  Such samples produce a zero-length logits tensor that causes
    ``selective_log_softmax`` to raise a shape-mismatch RuntimeError.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("compute_loss_func", None)
        # TRL 1.x DPOTrainer does not accept optimizer_cls_and_kwargs
        # (only SFTTrainer does); drop it silently.
        kwargs.pop("optimizer_cls_and_kwargs", None)
        super().__init__(*args, **kwargs)
        # Disable DataParallel: PP and DDP remain unaffected
        self.args._n_gpu = 1
        # Drop samples whose completion was fully truncated away.
        self.train_dataset = self._drop_empty_completions(self.train_dataset, "train")
        if self.eval_dataset is not None:
            if isinstance(self.eval_dataset, dict):
                self.eval_dataset = {
                    name: self._drop_empty_completions(ds, name)
                    for name, ds in self.eval_dataset.items()
                }
            else:
                self.eval_dataset = self._drop_empty_completions(self.eval_dataset, "eval")

    @staticmethod
    def _drop_empty_completions(dataset, split_name: str):
        """Remove tokenized samples where chosen_ids or rejected_ids is empty.

        This can happen when ``max_length`` is set and a very long prompt
        consumes the entire token budget, leaving no room for the completion.
        """
        if dataset is None:
            return dataset
        if "chosen_ids" not in dataset.column_names or "rejected_ids" not in dataset.column_names:
            return dataset

        before = len(dataset)
        dataset = dataset.filter(
            lambda ex: len(ex["chosen_ids"]) > 0 and len(ex["rejected_ids"]) > 0,
            desc=f"Filtering empty completions ({split_name})",
        )
        removed = before - len(dataset)
        if removed > 0:
            logger.warning(
                "DPOTrainerModule: removed %d sample(s) from '%s' split whose chosen_ids or "
                "rejected_ids were empty after tokenization/truncation. "
                "Consider increasing max_length or using truncation_mode='keep_end'.",
                removed,
                split_name,
            )
        return dataset
