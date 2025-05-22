# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import logging


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

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # define formatter
    ch.setFormatter(QEffFormatter())

    logger.addHandler(ch)
    return logger


# Define the logger object that can be used for logging purposes throughout the module.
logger = create_logger()


def create_ft_logger(log_file="finetune.log") -> logging.Logger:
    """
    Creates a logger object with Colored QEffFormatter.
    """
    logger = logging.getLogger("QEfficient")

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(QEffFormatter())
    logger.addHandler(ch)

    # create file handler and set level to debug
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(QEffFormatter())
    logger.addHandler(fh)

    return logger


# Define the logger object that can be used for logging purposes throughout the finetuning module.
ft_logger = create_ft_logger()
"""

class FT_Logger:
    def __init__(self, level=logging.INFO, log_file="finetune.log"):
        self.logger = logging.getLogger("QEfficient")
        self.logger.setLevel(level)
        self.level = level

        # Create handlers
        self.file_handler = logging.FileHandler(log_file)
        self.console_handler = logging.StreamHandler()

        self.file_handler.setFormatter(QEffFormatter())
        self.console_handler.setFormatter(QEffFormatter())

        # Add handlers to the logger
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)

    def get_logger(self):
        return self.logger
        
    def raise_valueerror(self, message):
        self.logger.error(message)
        raise ValueError(message)

    def raise_runtimeerror(self, message):
        self.logger.error(message)
        raise RuntimeError(message)
        
    def raise_filenotfounderror(self, message):
        self.logger.error(message)
        raise FileNotFoundError(message)

ft_logger = FT_Logger().get_logger()
"""
