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


class QEffFormatter(logging.Formatter):
    """
    Formatter class used to set colors for printing different logging levels of messages on console.
    """

    cyan: str = "\x1b[38;5;14m"
    yellow: str = "\x1b[33;20m"
    red: str = "\x1b[31;20m"
    bold_red: str = "\x1b[31;1m"
    reset: str = "\x1b[0m"
    common_format: str = "%(levelname)s - %(name)s - %(message)s"  # type: ignore
    format_with_line_info = "%(levelname)s - %(name)s - %(message)s  (%(filename)s:%(lineno)d)"  # type: ignore

    FORMATS = {
        logging.DEBUG: cyan + format_with_line_info + reset,
        logging.INFO: cyan + common_format + reset,
        logging.WARNING: yellow + common_format + reset,
        logging.ERROR: red + format_with_line_info + reset,
        logging.CRITICAL: bold_red + format_with_line_info + reset,
    }

    def format(self, record):
        """
        Overriding the base class method to Choose format based on log level.
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def create_logger() -> logging.Logger:
    """
    Creates a logger object with Colored QEffFormatter.
    """
    logger = logging.getLogger("QEfficient")

    # create console handler and set level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(QEffFormatter())
    logger.addHandler(ch)

    return logger


class CustomLogger(logging.Logger):
    def raise_runtimeerror(self, message):
        self.error(message)
        raise RuntimeError(message)

    def log_rank_zero(self, msg: str, level: int = logging.INFO) -> None:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank != 0:
            return
        self.log(level, msg, stacklevel=2)

    def prepare_dump_logs(self, dump_logs=False):
        if dump_logs:
            logs_path = os.path.join(ROOT_DIR, "logs")
            if not os.path.exists(logs_path):
                os.makedirs(logs_path, exist_ok=True)
            file_name = f"log-file-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" + ".txt"
            log_file = os.path.join(logs_path, file_name)

            # create file handler and set level
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
            fh.setFormatter(formatter)
            logger.addHandler(fh)


logging.setLoggerClass(CustomLogger)

# Define the logger object that can be used for logging purposes throughout the module.
logger = create_logger()
