# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import logging
import os
from datetime import datetime

import torch.distributed as dist

from QEfficient.utils.constants import ROOT_DIR


class FTLogger:
    def __init__(self, level=logging.DEBUG):
        self.logger = logging.getLogger("QEfficient")
        if not getattr(self.logger, "_custom_methods_added", False):
            self._bind_custom_methods()
            self.logger._custom_methods_added = True  # Prevent adding handlers/methods twice

    def _bind_custom_methods(self):
        def raise_runtimeerror(message):
            self.logger.error(message)
            raise RuntimeError(message)

        def log_rank_zero(msg: str, level: int = logging.INFO):
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            if rank != 0:
                return
            self.logger.log(level, msg, stacklevel=2)

        def prepare_dump_logs(dump_logs=False, level=logging.INFO):
            if dump_logs:
                logs_path = os.path.join(ROOT_DIR, "logs")
                if not os.path.exists(logs_path):
                    os.makedirs(logs_path, exist_ok=True)
                file_name = f"log-file-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" + ".txt"
                log_file = os.path.join(logs_path, file_name)

                fh = logging.FileHandler(log_file)
                fh.setLevel(level)
                formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)

        self.logger.raise_runtimeerror = raise_runtimeerror
        self.logger.log_rank_zero = log_rank_zero
        self.logger.prepare_dump_logs = prepare_dump_logs

    def get_logger(self):
        return self.logger


logger = FTLogger().get_logger()
