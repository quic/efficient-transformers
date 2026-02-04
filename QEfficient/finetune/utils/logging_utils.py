# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import logging
import os
from datetime import datetime

from QEfficient.finetune.utils.helper import get_node_rank, is_rank_zero


class FTLogger:
    def __init__(self):
        self.logger = logging.getLogger("QEfficient")
        if not getattr(self.logger, "_custom_methods_added", False):
            self._bind_custom_methods()
            self.logger._custom_methods_added = True  # Prevent adding handlers/methods twice

    def _bind_custom_methods(self):
        def raise_error(message, errortype=RuntimeError):
            self.logger.error(message)
            raise errortype(message)

        def log_rank_zero(msg: str, level: int = logging.INFO):
            if is_rank_zero():
                self.logger.log(level, msg, stacklevel=2)

        def prepare_for_logs(output_path, dump_logs=False, level=logging.INFO):
            self.logger.setLevel(level)
            if dump_logs:
                node_rank = get_node_rank()
                output_path = f"{output_path}_node_rank_{node_rank}"
                logs_path = os.path.join(output_path, "logs")
                if not os.path.exists(logs_path):
                    os.makedirs(logs_path, exist_ok=True)
                file_name = f"log-file-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" + ".txt"
                log_file = os.path.join(logs_path, file_name)

                fh = logging.FileHandler(log_file)
                fh.setLevel(level)
                formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)

        self.logger.raise_error = raise_error
        self.logger.log_rank_zero = log_rank_zero
        self.logger.prepare_for_logs = prepare_for_logs

    def get_logger(self):
        return self.logger


logger = FTLogger().get_logger()
