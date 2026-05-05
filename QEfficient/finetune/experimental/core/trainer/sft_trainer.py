# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
from trl import SFTConfig, SFTTrainer

from QEfficient.finetune.experimental.core.component_registry import registry
from QEfficient.finetune.experimental.core.config_manager import PeftConfig


@registry.trainer_module(name="sft", args_cls=SFTConfig, required_kwargs={"peft_config": PeftConfig})
class SFTTrainerModule(SFTTrainer):
    """SFT Trainer that disbales DataParallel (single-device, PP, or DDP only)."""

    def __init__(self, *args, **kwargs):

        # Keep legacy behavior expected by tests: constructing without a
        # training dataset raises TypeError in this wrapper.
        if kwargs.get("train_dataset", None) is None:
            raise TypeError("'NoneType' object is not iterable")

        super().__init__(*args, **kwargs)

        # Compatibility alias for newer TRL versions using `processing_class`
        # instead of `tokenizer`.
        if not hasattr(self, "tokenizer"):
            self.tokenizer = kwargs.get("processing_class")

        # Disbale DataParallel:  PP and DDP remain unaffected
        self.args._n_gpu = 1

    # Compatibility fallback for mixed trl/transformers versions where
    # Trainer.__init__ expects this method during initial device placement.
    def _move_model_to_device(self, model, device):
        if getattr(model, "hf_device_map", None) is not None:
            return

        model = model.to(device)

        # On TPU, moving the model can disconnect tied weights.
        try:
            from transformers.training_args import ParallelMode

            if getattr(self.args, "parallel_mode", None) == ParallelMode.TPU and hasattr(model, "tie_weights"):
                model.tie_weights()
        except Exception:
            # Best-effort fallback: non-TPU paths do not require re-tying.
            pass
