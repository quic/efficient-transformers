# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
from typing import Any, Dict, Optional

from transformers import (
    DefaultFlowCallback,
    EarlyStoppingCallback,
    PrinterCallback,
    ProgressCallback,
    TrainingArguments,
)
from transformers.integrations.integration_utils import TensorBoardCallback
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from QEfficient.finetune.experimental.core.component_registry import ComponentFactory, registry
from QEfficient.finetune.experimental.core.utils.profiler_utils import (
    get_op_verifier_ctx,
    init_qaic_profiling,
    stop_qaic_profiling,
)

registry.callback("early_stopping")(EarlyStoppingCallback)
registry.callback("printer")(PrinterCallback)
registry.callback("default_flow")(DefaultFlowCallback)
registry.callback("tensorboard")(TensorBoardCallback)


@registry.callback("enhanced_progressbar")
class EnhancedProgressCallback(ProgressCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation.
    You can modify `max_str_len` to control how long strings are truncated when logging.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the callback with optional max_str_len parameter to control string truncation length.

        Args:
            max_str_len (`int`):
                Maximum length of strings to display in logs.
                Longer strings will be truncated with a message.
        """
        super().__init__(*args, **kwargs)

    def on_train_begin(self, args, state, control, **kwargs):
        """Set progress bar description at the start of training."""
        super().on_train_begin(args, state, control, **kwargs)
        if self.training_bar is not None:
            self.training_bar.set_description("Training Progress")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Override the default `on_log` behavior during training to display
        the current epoch number, loss, and learning rate in the logs.
        """
        if state.is_world_process_zero and self.training_bar is not None:
            # make a shallow copy of logs so we can mutate the fields copied
            # but avoid doing any value pickling.
            shallow_logs = {}
            for k, v in logs.items():
                if isinstance(v, str) and len(v) > self.max_str_len:
                    shallow_logs[k] = (
                        f"[String too long to display, length: {len(v)} > {self.max_str_len}. "
                        "Consider increasing `max_str_len` if needed.]"
                    )
                else:
                    shallow_logs[k] = v
            _ = shallow_logs.pop("total_flos", None)
            # round numbers so that it looks better in console
            if "epoch" in shallow_logs:
                shallow_logs["epoch"] = round(shallow_logs["epoch"], 2)

            updated_dict = {}
            if "epoch" in shallow_logs:
                updated_dict["epoch"] = shallow_logs["epoch"]
            if "loss" in shallow_logs:
                updated_dict["loss"] = shallow_logs["loss"]
            if "learning_rate" in shallow_logs:
                updated_dict["lr"] = shallow_logs["learning_rate"]
            self.training_bar.set_postfix(updated_dict)


@registry.callback("json_logger")
class JSONLoggerCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that logs training and evaluation metrics to a JSON file.
    """

    def __init__(self, log_path=None, *args, **kwargs):
        """
        Initialize the callback with the path to the JSON log file.

        Args:
            log_path (`str`):
                Path to the jsonl file where logs will be saved.
        """
        super().__init__(*args, **kwargs)
        if log_path is None:
            log_path = os.path.join(os.environ.get("OUTPUT_DIR", "./"), "training_logs.jsonl")
        self.log_path = log_path
        # Ensure the log file is created and empty
        with open(self.log_path, "w") as _:
            pass

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict] = None,
        **kwargs,
    ):
        """Append sanitized log metrics (including global_step) to a JSONL file."""
        if logs is None:
            return
        logs.pop("entropy", None)
        logs.pop("mean_token_accuracy", None)
        if state.global_step:
            logs["global_step"] = state.global_step
        if logs is not None:
            with open(self.log_path, "a") as f:
                json_line = json.dumps(logs, separators=(",", ":"))
                f.write(json_line + "\n")


@registry.callback("qaic_profiler_callback")
class QAICProfilerCallback(TrainerCallback):
    """Callback to profile QAIC devices over a specified training step range."""

    def __init__(self, *args, **kwargs):
        """
        Initialize QAIC profiler settings (start/end steps and target device IDs).
        """

        self.start_step = kwargs.get("start_step", -1)
        self.end_step = kwargs.get("end_step", -1)
        self.device_ids = kwargs.get("device_ids", [0])

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        if state.global_step == self.start_step:
            for device_id in self.device_ids:
                init_qaic_profiling(True, f"qaic:{device_id}")
        elif state.global_step == self.end_step:
            for device_id in self.device_ids:
                stop_qaic_profiling(True, f"qaic:{device_id}")


@registry.callback("qaic_op_by_op_verifier_callback")
class QAICOpByOpVerifierCallback(TrainerCallback):
    """Callback to verify QAIC operations step-by-step during a specified training range."""

    def __init__(self, *args, **kwargs):
        """ "
        Initialize QAIC Op-by-Op verifier callback with profiling and tolerance settings.
        """
        self.start_step = kwargs.get("start_step", -1)
        self.end_step = kwargs.get("end_step", -1)
        self.trace_dir = kwargs.get("trace_dir", "qaic_op_by_op_traces")
        self.atol = kwargs.get("atol", 1e-1)
        self.rtol = kwargs.get("rtol", 1e-5)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        if self.start_step <= state.global_step < self.end_step:
            self.op_verifier_ctx_step = get_op_verifier_ctx(
                use_op_by_op_verifier=True,
                device_type="qaic",
                dump_dir=self.trace_dir,
                step=state.global_step,
                atol=self.atol,
                rtol=self.rtol,
            )
            self.op_verifier_ctx_step.__enter__()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        if self.start_step <= state.global_step < self.end_step:
            if self.op_verifier_ctx_step is not None:
                self.op_verifier_ctx_step.__exit__(None, None, None)


def replace_progress_callback(trainer: Any, callbacks: list[Any], logger: Any = None) -> None:
    """
    Replace default ProgressCallback with EnhancedProgressCallback if not already present.

    Args:
        trainer: Trainer instance
        callbacks: List of callbacks already added
        logger: Optional logger instance for warning messages
    """
    # Check if EnhancedProgressCallback is already in callbacks
    has_enhanced = any(callback.__class__.__name__ == "EnhancedProgressCallback" for callback in callbacks)

    if not has_enhanced:
        try:
            # Remove default ProgressCallback if present
            trainer.remove_callback(ProgressCallback)
        except (AttributeError, ValueError) as e:
            # Callback not present or method doesn't exist, continue
            if logger:
                logger.log_rank_zero(
                    f"Debug: Could not remove default ProgressCallback: {e}. This is expected if callback is not present.",
                    level="debug",
                )
            pass

        try:
            # Add EnhancedProgressCallback
            enhanced_callback = ComponentFactory.create_callback("enhanced_progressbar")
            trainer.add_callback(enhanced_callback)
        except Exception as e:
            if logger:
                logger.log_rank_zero(f"Warning: Could not add enhanced progress callback: {e}", level="warning")
            else:
                import warnings

                warnings.warn(f"Could not add enhanced progress callback: {e}")
