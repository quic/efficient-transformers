# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import logging
import sys
from pathlib import Path
from typing import Optional

from transformers.utils.logging import get_logger as hf_get_logger

from QEfficient.finetune.experimental.core.utils.dist_utils import get_local_rank

# -----------------------------------------------------------------------------
# Logger usage:
# Initialize logger:
#   logger = Logger("my_logger", log_file="logs/output.log", level=logging.DEBUG)
# Log messages:
#   logger.info("This is an info message")
#   logger.error("This is an error message")
#   logger.log_rank_zero("This message is logged only on rank 0")
#   logger.log_exception("An error occurred", exception, raise_exception=False)
# Attach file handler later if needed:
#   logger.prepare_for_logs(output_dir="logs", log_level="DEBUG")
# -----------------------------------------------------------------------------


class Logger:
    """Custom logger with console and file logging capabilities."""

    def __init__(
        self,
        name: str = "transformers",  # We are using "transformers" as default to align with HF logs
        log_file: Optional[str] = None,
        level: int = logging.INFO,
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name
            log_file: Path to log file (if None, log only to console)
            level: Logging level
        """
        self.logger = hf_get_logger(name)
        self.logger.setLevel(level)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Create formatter
        self.formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)

        # File handler (if log_file is provided)
        if log_file:
            # Create directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)

    def log_rank_zero(self, message: str, level: int = logging.INFO) -> None:
        """
        Log message only on rank 0 process.

        Args:
            message: Message to log
            level: Logging level
        """
        if get_local_rank() == 0:
            self.logger.log(level, message)

    def log_exception(self, message: str, exception: Exception, raise_exception: bool = True) -> None:
        """
        Log exception message and optionally raise the exception.

        Args:
            message: Custom message to log
            exception: Exception to log
            raise_exception: Whether to raise the exception after logging
        """
        error_message = f"{message}: {str(exception)}"
        self.logger.error(error_message)

        if raise_exception:
            raise exception

    def prepare_for_logs(self, output_dir: Optional[str] = None, log_level: str = "INFO") -> None:
        """
        Prepare existing logger to log to both console and file with specified
        output directory and log level.

        Args:
            output_dir: Output directory for logs
            log_level: Logging level as string
        """
        # Convert string log level to logging constant
        level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(level)

        # Update existing handlers' levels
        for handler in self.logger.handlers:
            handler.setLevel(level)

        # Add file handler if saving metrics
        if output_dir:
            log_file = Path(output_dir) / "training.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            # Check if file handler already exists
            file_handler_exists = any(isinstance(handler, logging.FileHandler) for handler in self.logger.handlers)

            if not file_handler_exists:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(level)
                file_handler.setFormatter(self.formatter)
                self.logger.addHandler(file_handler)


# Global logger instance
_logger: Optional[Logger] = None


def get_logger(log_file: Optional[str] = None) -> Logger:
    """
    Get or create a logger instance.

    Args:
        log_file: Path to log file (if None, log only to console)

    Returns:
        Logger instance
    """
    global _logger
    if _logger is None:
        _logger = Logger(log_file=log_file)
    return _logger
