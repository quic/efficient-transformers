# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import logging
import os
from datetime import datetime
from typing import Optional

from concurrent_log_handler import ConcurrentRotatingFileHandler


class QEFFLogger:
    _instance: Optional[logging.Logger] = None

    def __init__(self, loglevel: Optional[str] = "INFO"):
        if QEFFLogger._instance is None:
            self.loglevel = loglevel
            self.logger = self._initialize_logger()
            QEFFLogger._instance = self.logger

    def _get_formatter(self) -> logging.Formatter:
        return logging.Formatter(
            "[%(asctime)s:%(levelname)s:%(threadName)s:%(filename)s:%(lineno)d:%(funcName)s()]:%(message)s"
        )

    def _initialize_logger(self) -> logging.Logger:
        # Define the hidden log directory path
        log_dir = os.path.expanduser("~/.cache/.log")
        os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # Create a timestamped log file in the hidden directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logfile = os.path.join(log_dir, f"QEFF_{timestamp}.log")

        numeric_level = getattr(logging, self.loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {self.loglevel}")

        logger = logging.getLogger("QEFF_LOGGER")
        logger.setLevel(numeric_level)

        if not logger.handlers:
            handler = ConcurrentRotatingFileHandler(logfile, maxBytes=5 * 1024 * 1024, backupCount=15)
            handler.setFormatter(self._get_formatter())
            logger.addHandler(handler)

        return logger

    @classmethod
    def get_logger(cls, loglevel: Optional[str] = "INFO") -> logging.Logger:
        if cls._instance is None:
            cls(loglevel)
        return cls._instance

    @classmethod
    def set_loglevel(cls, loglevel: Optional[str] = "INFO"):
        if cls._instance is None:
            raise RuntimeError("Logger has not been initialized yet. Call get_logger() first.")

        numeric_level = getattr(logging, loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {loglevel}")

        cls._instance.setLevel(numeric_level)

    @classmethod
    def close_logger(cls):
        if cls._instance:
            handlers = cls._instance.handlers[:]
            for handler in handlers:
                handler.close()
                cls._instance.removeHandler(handler)
            cls._instance = None
