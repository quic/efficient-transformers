# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import logging
import os
import queue
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional


class JSONNamespaceFormatter(logging.Formatter):
    """
    Custom formatter to output log records in JSON format with metadata.

    Methods:
        format(record): Formats a log record into a JSON string.

    Parameters:
        record (logging.LogRecord): The log record to format.

    Returns:
        str: JSON-formatted log string.
    """

    def format(self, record):
        log_record = {
            "date": datetime.fromtimestamp(record.created).strftime("%Y-%m-%d"),
            "time": datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
            "level": record.levelname,
            "namespace": getattr(record, "namespace", "default"),
            "file": record.filename,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        return json.dumps(log_record)


class QEFFLoggerThread(threading.Thread):
    """
    Background thread to handle logging asynchronously using a queue.

    Attributes:
        logger (logging.Logger): Logger instance to handle log records.
        log_queue (queue.Queue): Queue from which log records are consumed.
        running (bool): Flag to control thread execution.
    """

    def __init__(self, logger, log_queue):
        """
        Initialize the logging thread.

        Parameters:
            logger (logging.Logger): Logger instance.
            log_queue (queue.Queue): Queue for log records.
        """
        super().__init__(daemon=True)
        self.logger = logger
        self.log_queue = log_queue
        self.running = True

    def run(self):
        """
        Continuously process log records from the queue and pass them to the logger.
        """
        while self.running:
            try:
                record = self.log_queue.get(timeout=1)
                self.logger.handle(record)
            except queue.Empty:
                continue

    def stop(self):
        """
        Stop the logging thread gracefully.
        """
        self.running = False


class QEFFLogger:
    """
    Singleton logger class for structured logging with namespace support.

    Class Attributes:
        _instance (Optional[logging.Logger]): Singleton logger instance.
        _logfile (Optional[str]): Path to the log file.
        _log_queue (queue.Queue): Queue for asynchronous logging.
        _logger_thread (Optional[QEFFLoggerThread]): Background logging thread.
    """

    _instance: Optional[logging.Logger] = None
    _logfile: Optional[str] = None
    _log_queue: queue.Queue = queue.Queue()
    _logger_thread: Optional[QEFFLoggerThread] = None

    def __init__(self, loglevel: Optional[str] = "INFO", log_path: Optional[str] = None):
        """
        Initialize the logger instance with specified log level and path.

        Parameters:
            loglevel (str): Logging level (e.g., "INFO", "DEBUG").
            log_path (str): Optional path to the log file.
        """
        if QEFFLogger._instance is None:
            self.loglevel = loglevel
            self.log_path = log_path
            self.logger = self._initialize_logger()
            QEFFLogger._instance = self.logger
            QEFFLogger._logger_thread = QEFFLoggerThread(self.logger, QEFFLogger._log_queue)
            QEFFLogger._logger_thread.start()

    def _initialize_logger(self) -> logging.Logger:
        """
        Set up the logger with rotating file handler and JSON formatter.

        Returns:
            logging.Logger: Configured logger instance.
        """
        if self.log_path is None:
            log_dir = os.path.expanduser("~/.cache/qefficient_logs")
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = os.path.join(log_dir, f"QEFF_{timestamp}.log")

        QEFFLogger._logfile = self.log_path

        numeric_level = getattr(logging, self.loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {self.loglevel}")

        logger = logging.getLogger("QEFF_LOGGER")
        logger.setLevel(numeric_level)

        if not logger.handlers:
            handler = RotatingFileHandler(self.log_path, maxBytes=5 * 1024 * 1024, backupCount=10)
            handler.setFormatter(JSONNamespaceFormatter())
            logger.addHandler(handler)

        return logger

    @classmethod
    def get_logger(
        cls, namespace: str, loglevel: Optional[str] = "INFO", log_path: Optional[str] = None
    ) -> logging.Logger:
        """
        Retrieve a logger adapter with a specific namespace.

        Parameters:
            namespace (str): Logical grouping for the log.
            loglevel (str): Logging level.
            log_path (str): Optional path to the log file.

        Returns:
            logging.Logger: Logger adapter with namespace.
        """
        if cls._instance is None:
            cls(loglevel, log_path)
        return logging.LoggerAdapter(cls._instance, {"namespace": namespace})

    @classmethod
    def log(cls, level: str, namespace: str, msg: str, fn: str = "", lno: int = 0, func: str = ""):
        """
        Log a message with specified level and metadata.

        Parameters:
            level (str): Logging level (e.g., "INFO", "ERROR").
            namespace (str): Logical grouping for the log.
            msg (str): Log message.
            fn (str): Filename where the log is generated.
            lno (int): Line number in the file.
            func (str): Function name.
        """
        if cls._instance is None:
            raise RuntimeError("Logger has not been initialized. Call get_logger() first.")

        level_num = getattr(logging, level.upper(), None)
        if not isinstance(level_num, int):
            raise ValueError(f"Invalid log level: {level}")

        record = cls._instance.makeRecord(
            name="QEFF_LOGGER",
            level=level_num,
            fn=fn,
            lno=lno,
            msg=msg,
            args=(),
            exc_info=None,
            func=func,
            extra={"namespace": namespace},
        )
        cls._log_queue.put(record)

    @classmethod
    def set_loglevel(cls, loglevel: Optional[str] = "INFO"):
        """
        Update the log level of the logger.

        Parameters:
            loglevel (str): New log level to set.
        """
        if cls._instance is None:
            raise RuntimeError("Logger has not been initialized yet. Call get_logger() first.")

        numeric_level = getattr(logging, loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {loglevel}")

        cls._instance.setLevel(numeric_level)

    @classmethod
    def close_logger(cls):
        """
        Gracefully shut down the logger and its thread.
        """
        if cls._logger_thread:
            cls._logger_thread.stop()
            cls._logger_thread.join()
            cls._logger_thread = None

        if cls._instance:
            handlers = cls._instance.handlers[:]
            for handler in handlers:
                handler.close()
                cls._instance.removeHandler(handler)
            cls._instance = None
            cls._logfile = None
