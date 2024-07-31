# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from transformers import Trainer

from .data_utils import QEffDataManager
from .model_manager import QEffModelManager


class QEffTrainer(Trainer):
    def __init__(self, model_manager: QEffModelManager, data_manager: QEffDataManager, **kwargs):
        self.model_manager = model_manager
        self.data_manager = data_manager
        super().__init__(**kwargs)

    def inner_training_loop():
        # Implement custom inner training loop
        pass

    def train(self, resume_from_checkpoint: str = None, trial=None, **kwargs):
        # Implement custom training loop here
        pass

    def evaluate(self, eval_dataset=None, **kwargs):
        # Implement evaluation here
        pass

    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        # Implement model saving here
        pass
